# coding: utf-8


class ImpalaUtils:

    @staticmethod
    def get_impala_connector(cfg: dict):
        """ Generates an Impala connector

        :param cfg: Impala connection configuration
        :return: Impala connector
        """
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        from impala.dbapi import connect

        # noinspection PyCallingNonCallable
        return connect(**cfg)
